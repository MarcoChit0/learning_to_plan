; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 226971

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child285 - child
    bread159 - bread-portion
    content153 - content-portion
    tray237 tray150 tray36 tray226 - tray
    table374 table116 table288 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray237 kitchen)
     (at tray150 kitchen)
     (at tray36 kitchen)
     (at tray226 kitchen)
     (at_kitchen_bread bread159)
     (at_kitchen_content content153)
     (not_allergic_gluten child285)
     (waiting child285 table116)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child285)
    )
  )
)
