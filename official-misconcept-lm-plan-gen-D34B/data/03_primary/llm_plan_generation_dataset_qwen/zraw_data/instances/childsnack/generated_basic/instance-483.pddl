; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 964032

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child53 child398 - child
    bread7 bread468 - bread-portion
    content349 content208 - content-portion
    tray201 tray322 - tray
    table39 table71 table185 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray201 kitchen)
     (at tray322 kitchen)
     (at_kitchen_bread bread7)
     (at_kitchen_bread bread468)
     (at_kitchen_content content349)
     (at_kitchen_content content208)
     (not_allergic_gluten child53)
     (not_allergic_gluten child398)
     (waiting child53 table71)
     (waiting child398 table185)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child53)
     (served child398)
    )
  )
)
