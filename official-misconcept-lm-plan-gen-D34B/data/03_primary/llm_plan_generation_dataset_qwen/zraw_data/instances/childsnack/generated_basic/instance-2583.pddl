; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 457523

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child100 - child
    bread159 - bread-portion
    content166 - content-portion
    tray270 tray442 tray165 - tray
    table403 table265 table398 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray270 kitchen)
     (at tray442 kitchen)
     (at tray165 kitchen)
     (at_kitchen_bread bread159)
     (at_kitchen_content content166)
     (not_allergic_gluten child100)
     (waiting child100 table398)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child100)
    )
  )
)
