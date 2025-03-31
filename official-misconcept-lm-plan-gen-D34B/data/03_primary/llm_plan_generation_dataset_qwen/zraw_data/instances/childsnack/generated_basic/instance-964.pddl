; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 446644

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child158 child31 child449 - child
    bread353 bread481 bread238 - bread-portion
    content74 content139 content493 - content-portion
    tray174 tray282 tray170 tray323 - tray
    table137 table72 table192 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray174 kitchen)
     (at tray282 kitchen)
     (at tray170 kitchen)
     (at tray323 kitchen)
     (at_kitchen_bread bread353)
     (at_kitchen_bread bread481)
     (at_kitchen_bread bread238)
     (at_kitchen_content content74)
     (at_kitchen_content content139)
     (at_kitchen_content content493)
     (not_allergic_gluten child31)
     (not_allergic_gluten child158)
     (not_allergic_gluten child449)
     (waiting child158 table72)
     (waiting child31 table192)
     (waiting child449 table72)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child158)
     (served child31)
     (served child449)
    )
  )
)
