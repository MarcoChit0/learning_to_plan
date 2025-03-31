; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 785353

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child63 child467 child405 - child
    bread492 bread409 bread78 - bread-portion
    content113 content78 content180 - content-portion
    tray231 tray376 tray454 tray267 - tray
    table65 table386 table120 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray231 kitchen)
     (at tray376 kitchen)
     (at tray454 kitchen)
     (at tray267 kitchen)
     (at_kitchen_bread bread492)
     (at_kitchen_bread bread409)
     (at_kitchen_bread bread78)
     (at_kitchen_content content113)
     (at_kitchen_content content78)
     (at_kitchen_content content180)
     (not_allergic_gluten child63)
     (not_allergic_gluten child467)
     (not_allergic_gluten child405)
     (waiting child63 table120)
     (waiting child467 table386)
     (waiting child405 table386)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child63)
     (served child467)
     (served child405)
    )
  )
)
