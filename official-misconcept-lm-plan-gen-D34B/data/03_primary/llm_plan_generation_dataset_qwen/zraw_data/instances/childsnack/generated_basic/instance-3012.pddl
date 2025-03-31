; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 440043

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child485 child443 - child
    bread77 bread127 - bread-portion
    content183 content497 - content-portion
    tray484 tray44 tray203 tray161 - tray
    table187 table210 table26 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray484 kitchen)
     (at tray44 kitchen)
     (at tray203 kitchen)
     (at tray161 kitchen)
     (at_kitchen_bread bread77)
     (at_kitchen_bread bread127)
     (at_kitchen_content content183)
     (at_kitchen_content content497)
     (not_allergic_gluten child443)
     (not_allergic_gluten child485)
     (waiting child485 table210)
     (waiting child443 table210)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child485)
     (served child443)
    )
  )
)
