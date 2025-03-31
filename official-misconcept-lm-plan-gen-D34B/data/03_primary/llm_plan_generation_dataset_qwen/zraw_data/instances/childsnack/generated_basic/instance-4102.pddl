; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 935103

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child236 - child
    bread159 - bread-portion
    content16 - content-portion
    tray289 tray425 tray62 - tray
    table164 table202 table444 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray289 kitchen)
     (at tray425 kitchen)
     (at tray62 kitchen)
     (at_kitchen_bread bread159)
     (at_kitchen_content content16)
     (not_allergic_gluten child236)
     (waiting child236 table164)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child236)
    )
  )
)
