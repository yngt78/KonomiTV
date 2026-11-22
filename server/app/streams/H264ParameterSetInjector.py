"""H.264 の SPS/PPS を I フレームごとに注入するヘルパー

Raspberry Pi のハードウェアエンコーダー (bcm2835-codec / h264_v4l2m2m) は
SPS/PPS (デコードに必須のパラメーターセット) をストリームの先頭でしか出力しない。

KonomiTV は 1 本のエンコード出力を複数のクライアントで共有するため、
すでに再生中のストリームへ後から接続したクライアントは SPS/PPS を受け取れず、
映像をデコードできない (音声だけ再生される)。再読み込みや一時的な切断からの
再接続でも同じことが起きる。

x264 や Intel QSV は I フレームごとに SPS/PPS を繰り返すため、この問題は起きない。
ここでは ffmpeg の出力 (MPEG-TS) をパケット単位で書き換え、
x264 / QSV と同じ形 (I フレームの PES の先頭に SPS/PPS) にそろえる。
エンコーダーの差し替えや再エンコードは行わない。
"""
from __future__ import annotations

from app import logging


class H264ParameterSetInjector:
    """ffmpeg の出力 TS をパケット単位で受け取り、I フレームに SPS/PPS を注入する"""

    PACKET_SIZE = 188

    def __init__(self) -> None:
        self._pmt_pid: int | None = None
        self._video_pid: int | None = None
        self._sps: bytes | None = None
        self._pps: bytes | None = None
        # 映像 PID の連続性カウンタ
        ## SPS/PPS を挿入するとパケット数が増えてカウンタがずれるため、こちらで振り直す
        self._continuity_counter: int | None = None
        # 現在処理中の映像 PES の TS パケット (I フレームの PES を書き換えるためにまとめて持つ)
        self._pes_packets: list[bytes] = []
        self._injected_count = 0
        self._logged_ready = False

    def process(self, packet: bytes) -> list[bytes]:
        """TS パケットを 1 つ受け取り、出力する TS パケットのリストを返す"""

        # TS パケットでないものはそのまま流す
        if len(packet) != self.PACKET_SIZE or packet[0] != 0x47:
            return [packet]

        pid = ((packet[1] & 0x1F) << 8) | packet[2]

        # PAT (PID 0) から PMT の PID を得る
        if pid == 0 and self._pmt_pid is None:
            pmt_pid = self._find_pmt_pid(packet)
            if pmt_pid is not None:
                self._pmt_pid = pmt_pid
            return [packet]

        # PMT から映像 PID を得る
        if self._pmt_pid is not None and pid == self._pmt_pid and self._video_pid is None:
            video_pid = self._find_video_pid(packet)
            if video_pid is not None:
                self._video_pid = video_pid
            return [packet]

        if self._video_pid is None or pid != self._video_pid:
            return [packet]

        # 映像 PES の先頭が来たら、直前の PES をまとめて出力する
        output: list[bytes] = []
        if (packet[1] & 0x40) != 0 and len(self._pes_packets) > 0:
            output.extend(self._flush_pes())
        self._pes_packets.append(packet)
        return output

    def flush(self) -> list[bytes]:
        """残っている PES を出力する (ストリーム終了時用)"""

        output: list[bytes] = []
        if len(self._pes_packets) > 0:
            output.extend(self._flush_pes())
        return output

    def _flush_pes(self) -> list[bytes]:
        """まとめてある 1 本の映像 PES を出力する (必要なら SPS/PPS を注入する)"""

        packets = self._pes_packets
        self._pes_packets = []

        header, head_es = self._split_pes(packets[0])
        if header is None:
            return self._restamp_continuity(packets)

        nal_types = self._nal_types(head_es)

        # ストリーム先頭の SPS/PPS を保持しておく
        if self._sps is None or self._pps is None:
            if 7 in nal_types or 8 in nal_types:
                self._capture_parameter_sets(packets, header)
            return self._restamp_continuity(packets)

        # I フレーム (スライスが IDR) で、まだ SPS/PPS を含んでいない PES の先頭に挿入する
        ## 自前で SPS/PPS を出しているエンコーダー (x264 / QSV) の出力はそのまま通す
        if 5 in nal_types and 7 not in nal_types:
            es = b''.join(self._es_fragments(packets, header))
            ## PES_packet_length は 0 (= 長さ不明) にする。映像ストリームでは許容されている。
            pes = header[0:4] + b'\x00\x00' + header[6:] + self._sps + self._pps + es
            self._injected_count += 1
            if self._logged_ready is False:
                self._logged_ready = True
                logging.info(
                    'H.264 の SPS/PPS を I フレームごとに注入します '
                    f'(SPS {len(self._sps)} bytes / PPS {len(self._pps)} bytes)'
                )
            return self._packetize(pes, self._pcr(packets[0]))

        return self._restamp_continuity(packets)

    def _restamp_continuity(self, packets: list[bytes]) -> list[bytes]:
        """映像 PID の連続性カウンタを振り直す (注入でパケット数が変わっても連続にする)"""

        output: list[bytes] = []
        for packet in packets:
            # アダプテーションフィールドだけでペイロードを持たないパケットはカウンタを進めない
            if self._payload(packet) is None:
                output.append(packet)
                continue
            if self._continuity_counter is None:
                self._continuity_counter = packet[3] & 0x0F
            packet = packet[0:3] + bytes([(packet[3] & 0xF0) | self._continuity_counter]) + packet[4:]
            self._continuity_counter = (self._continuity_counter + 1) & 0x0F
            output.append(packet)
        return output

    def _packetize(self, pes: bytes, pcr: bytes | None = None) -> list[bytes]:
        """PES を TS パケットに分割する

        pcr を渡すと、先頭パケットのアダプテーションフィールドに転記する
        (書き換えで PCR を落とすと PCR の間隔が開きすぎるため)
        """

        output: list[bytes] = []
        offset = 0
        is_first = True
        pid = self._video_pid if self._video_pid is not None else 0
        while offset < len(pes):

            # PCR を持つパケットはアダプテーションフィールドの分だけペイロードが減る
            has_pcr = is_first is True and pcr is not None
            capacity = 176 if has_pcr is True else 184
            payload = pes[offset:offset + capacity]
            offset += len(payload)

            header = bytearray(4)
            header[0] = 0x47
            header[1] = (0x40 if is_first is True else 0x00) | ((pid >> 8) & 0x1F)
            header[2] = pid & 0xFF
            counter = self._continuity_counter if self._continuity_counter is not None else 0
            adaptation_length = 184 - len(payload)

            if adaptation_length == 0:
                header[3] = 0x10 | counter
                output.append(bytes(header) + payload)
            else:
                header[3] = 0x30 | counter
                field_length = adaptation_length - 1
                if field_length == 0:
                    field = b'\x00'
                else:
                    flags = 0x10 if has_pcr is True else 0x00
                    body = pcr if has_pcr is True and pcr is not None else b''
                    stuffing = field_length - 1 - len(body)
                    field = bytes([field_length, flags]) + body + bytes(max(0, stuffing))
                output.append(bytes(header) + field + payload)

            if self._continuity_counter is not None:
                self._continuity_counter = (self._continuity_counter + 1) & 0x0F
            is_first = False

        return output

    def _capture_parameter_sets(self, packets: list[bytes], header: bytes) -> None:
        """PES の中から SPS/PPS を取り出して保持する"""

        es = b''.join(self._es_fragments(packets, header))
        starts: list[int] = []
        offset = 0
        while True:
            index = es.find(b'\x00\x00\x01', offset)
            if index < 0:
                break
            starts.append(index)
            offset = index + 3

        for position, index in enumerate(starts):
            nal_type = es[index + 3] & 0x1F
            end = starts[position + 1] if position + 1 < len(starts) else len(es)
            if nal_type == 7 and self._sps is None:
                self._sps = b'\x00\x00\x00\x01' + es[index + 3:end]
            elif nal_type == 8 and self._pps is None:
                self._pps = b'\x00\x00\x00\x01' + es[index + 3:end]

    def _es_fragments(self, packets: list[bytes], header: bytes) -> list[bytes]:
        """PES の各 TS パケットから ES (エレメンタリーストリーム) を取り出す"""

        fragments: list[bytes] = []
        first_payload = self._payload(packets[0])
        if first_payload is not None:
            fragments.append(first_payload[len(header):])
        for packet in packets[1:]:
            payload = self._payload(packet)
            if payload is not None:
                fragments.append(payload)
        return fragments

    @staticmethod
    def _nal_types(es: bytes) -> set[int]:
        """ES に含まれる NAL ユニットの種別を集める"""

        nal_types: set[int] = set()
        offset = 0
        while True:
            index = es.find(b'\x00\x00\x01', offset)
            if index < 0 or index + 3 >= len(es):
                break
            nal_types.add(es[index + 3] & 0x1F)
            offset = index + 3
        return nal_types

    @staticmethod
    def _payload(packet: bytes) -> bytes | None:
        """TS パケットのペイロードを取り出す (アダプテーションフィールドを除く)"""

        adaptation_field_control = (packet[3] >> 4) & 0x03
        offset = 4
        if adaptation_field_control & 0x02:
            offset += 1 + packet[4]
        if not (adaptation_field_control & 0x01) or offset >= packet.__len__():
            return None
        return packet[offset:]

    @staticmethod
    def _pcr(packet: bytes) -> bytes | None:
        """アダプテーションフィールドの PCR (6 bytes) を取り出す"""

        adaptation_field_control = (packet[3] >> 4) & 0x03
        if not (adaptation_field_control & 0x02) or packet[4] < 7:
            return None
        if not (packet[5] & 0x10):
            return None
        return packet[6:12]

    @classmethod
    def _split_pes(cls, packet: bytes) -> tuple[bytes | None, bytes]:
        """PES のヘッダーと、そのパケットに入っている ES を返す"""

        payload = cls._payload(packet)
        if payload is None or len(payload) < 9 or payload[0:3] != b'\x00\x00\x01':
            return (None, b'')
        header_length = 9 + payload[8]
        if header_length > len(payload):
            return (None, b'')
        return (payload[0:header_length], payload[header_length:])

    @classmethod
    def _find_pmt_pid(cls, packet: bytes) -> int | None:
        """PAT から PMT の PID を求める"""

        if (packet[1] & 0x40) == 0:
            return None
        payload = cls._payload(packet)
        if payload is None or len(payload) < 1:
            return None
        section = payload[1 + payload[0]:]
        if len(section) < 12 or section[0] != 0x00:
            return None
        section_length = ((section[1] & 0x0F) << 8) | section[2]
        end = 3 + section_length - 4
        position = 8
        while position + 4 <= end:
            program_number = (section[position] << 8) | section[position + 1]
            pid = ((section[position + 2] & 0x1F) << 8) | section[position + 3]
            if program_number != 0:
                return pid
            position += 4
        return None

    @classmethod
    def _find_video_pid(cls, packet: bytes) -> int | None:
        """PMT から H.264 (stream_type 0x1B) の映像 PID を求める"""

        if (packet[1] & 0x40) == 0:
            return None
        payload = cls._payload(packet)
        if payload is None or len(payload) < 1:
            return None
        section = payload[1 + payload[0]:]
        if len(section) < 16 or section[0] != 0x02:
            return None
        section_length = ((section[1] & 0x0F) << 8) | section[2]
        end = 3 + section_length - 4
        position = 12 + (((section[10] & 0x0F) << 8) | section[11])
        while position + 5 <= end:
            stream_type = section[position]
            pid = ((section[position + 1] & 0x1F) << 8) | section[position + 2]
            es_info_length = ((section[position + 3] & 0x0F) << 8) | section[position + 4]
            if stream_type == 0x1B:
                return pid
            position += 5 + es_info_length
        return None
