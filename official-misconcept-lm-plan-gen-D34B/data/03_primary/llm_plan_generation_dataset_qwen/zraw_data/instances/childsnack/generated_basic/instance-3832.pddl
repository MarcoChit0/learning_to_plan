; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 419796

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child448 - child
    bread193 - bread-portion
    content127 - content-portion
    tray261 tray239 tray50 tray153 - tray
    table114 table134 table347 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray261 kitchen)
     (at tray239 kitchen)
     (at tray50 kitchen)
     (at tray153 kitchen)
     (at_kitchen_bread bread193)
     (at_kitchen_content content127)
     (not_allergic_gluten child448)
     (waiting child448 table134)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child448)
    )
  )
)
