; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 227823

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child146 child274 child119 - child
    bread430 bread442 bread55 - bread-portion
    content412 content57 content371 - content-portion
    tray160 - tray
    table212 table481 table245 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray160 kitchen)
     (at_kitchen_bread bread430)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread55)
     (at_kitchen_content content412)
     (at_kitchen_content content57)
     (at_kitchen_content content371)
     (not_allergic_gluten child274)
     (not_allergic_gluten child146)
     (not_allergic_gluten child119)
     (waiting child146 table481)
     (waiting child274 table212)
     (waiting child119 table212)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child146)
     (served child274)
     (served child119)
    )
  )
)
