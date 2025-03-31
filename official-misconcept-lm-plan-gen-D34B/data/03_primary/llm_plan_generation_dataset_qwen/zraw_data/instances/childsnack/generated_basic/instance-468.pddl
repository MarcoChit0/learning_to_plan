; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 412176

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child365 child261 child14 - child
    bread281 bread323 bread252 - bread-portion
    content403 content200 content328 - content-portion
    tray225 - tray
    table356 table385 table314 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray225 kitchen)
     (at_kitchen_bread bread281)
     (at_kitchen_bread bread323)
     (at_kitchen_bread bread252)
     (at_kitchen_content content403)
     (at_kitchen_content content200)
     (at_kitchen_content content328)
     (not_allergic_gluten child261)
     (not_allergic_gluten child365)
     (not_allergic_gluten child14)
     (waiting child365 table356)
     (waiting child261 table385)
     (waiting child14 table314)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child365)
     (served child261)
     (served child14)
    )
  )
)
