; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 995996

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child148 - child
    bread179 - bread-portion
    content373 - content-portion
    tray482 tray355 tray27 - tray
    table39 table160 table171 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray482 kitchen)
     (at tray355 kitchen)
     (at tray27 kitchen)
     (at_kitchen_bread bread179)
     (at_kitchen_content content373)
     (not_allergic_gluten child148)
     (waiting child148 table160)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child148)
    )
  )
)
