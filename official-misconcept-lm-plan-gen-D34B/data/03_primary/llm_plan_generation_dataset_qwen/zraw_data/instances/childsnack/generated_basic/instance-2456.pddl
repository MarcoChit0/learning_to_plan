; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 844180

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child200 child101 - child
    bread394 bread469 - bread-portion
    content373 content204 - content-portion
    tray311 tray431 - tray
    table400 table369 table343 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray311 kitchen)
     (at tray431 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_bread bread469)
     (at_kitchen_content content373)
     (at_kitchen_content content204)
     (not_allergic_gluten child200)
     (not_allergic_gluten child101)
     (waiting child200 table369)
     (waiting child101 table400)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child200)
     (served child101)
    )
  )
)
