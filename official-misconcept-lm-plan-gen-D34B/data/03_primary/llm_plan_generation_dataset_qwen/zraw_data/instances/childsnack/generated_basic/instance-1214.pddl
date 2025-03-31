; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 321837

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child316 child256 child17 - child
    bread407 bread100 bread435 - bread-portion
    content159 content286 content388 - content-portion
    tray230 tray128 tray248 - tray
    table388 table369 table95 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray230 kitchen)
     (at tray128 kitchen)
     (at tray248 kitchen)
     (at_kitchen_bread bread407)
     (at_kitchen_bread bread100)
     (at_kitchen_bread bread435)
     (at_kitchen_content content159)
     (at_kitchen_content content286)
     (at_kitchen_content content388)
     (not_allergic_gluten child17)
     (not_allergic_gluten child316)
     (not_allergic_gluten child256)
     (waiting child316 table388)
     (waiting child256 table95)
     (waiting child17 table388)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child316)
     (served child256)
     (served child17)
    )
  )
)
