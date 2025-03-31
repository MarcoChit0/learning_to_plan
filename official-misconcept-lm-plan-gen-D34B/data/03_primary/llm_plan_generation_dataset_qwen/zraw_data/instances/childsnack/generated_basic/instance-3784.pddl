; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 790144

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child310 - child
    bread271 - bread-portion
    content238 - content-portion
    tray54 - tray
    table98 table227 table167 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray54 kitchen)
     (at_kitchen_bread bread271)
     (at_kitchen_content content238)
     (not_allergic_gluten child310)
     (waiting child310 table98)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child310)
    )
  )
)
