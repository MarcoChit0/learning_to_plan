; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 190877

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child12 child256 - child
    bread240 bread194 - bread-portion
    content330 content132 - content-portion
    tray23 tray51 tray400 - tray
    table35 table169 table31 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray23 kitchen)
     (at tray51 kitchen)
     (at tray400 kitchen)
     (at_kitchen_bread bread240)
     (at_kitchen_bread bread194)
     (at_kitchen_content content330)
     (at_kitchen_content content132)
     (not_allergic_gluten child256)
     (not_allergic_gluten child12)
     (waiting child12 table169)
     (waiting child256 table35)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child12)
     (served child256)
    )
  )
)
