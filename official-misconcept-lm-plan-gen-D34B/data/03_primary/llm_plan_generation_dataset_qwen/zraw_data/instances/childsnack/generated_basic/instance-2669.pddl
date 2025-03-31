; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 339135

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child288 - child
    bread446 - bread-portion
    content99 - content-portion
    tray88 tray63 tray12 tray228 - tray
    table420 table245 table30 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray88 kitchen)
     (at tray63 kitchen)
     (at tray12 kitchen)
     (at tray228 kitchen)
     (at_kitchen_bread bread446)
     (at_kitchen_content content99)
     (not_allergic_gluten child288)
     (waiting child288 table420)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child288)
    )
  )
)
