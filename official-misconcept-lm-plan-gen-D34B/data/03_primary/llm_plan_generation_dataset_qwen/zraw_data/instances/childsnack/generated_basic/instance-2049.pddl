; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 862397

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child271 child212 - child
    bread392 bread396 - bread-portion
    content104 content141 - content-portion
    tray9 tray41 tray375 - tray
    table432 table91 table103 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray9 kitchen)
     (at tray41 kitchen)
     (at tray375 kitchen)
     (at_kitchen_bread bread392)
     (at_kitchen_bread bread396)
     (at_kitchen_content content104)
     (at_kitchen_content content141)
     (not_allergic_gluten child271)
     (not_allergic_gluten child212)
     (waiting child271 table432)
     (waiting child212 table432)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child271)
     (served child212)
    )
  )
)
