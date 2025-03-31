; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 462565

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child5 child491 child318 - child
    bread43 bread391 bread353 - bread-portion
    content59 content50 content254 - content-portion
    tray108 - tray
    table325 table336 table184 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray108 kitchen)
     (at_kitchen_bread bread43)
     (at_kitchen_bread bread391)
     (at_kitchen_bread bread353)
     (at_kitchen_content content59)
     (at_kitchen_content content50)
     (at_kitchen_content content254)
     (not_allergic_gluten child491)
     (not_allergic_gluten child5)
     (not_allergic_gluten child318)
     (waiting child5 table184)
     (waiting child491 table325)
     (waiting child318 table325)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child5)
     (served child491)
     (served child318)
    )
  )
)
