; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 419454

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child251 child266 child49 - child
    bread33 bread202 bread391 - bread-portion
    content65 content196 content470 - content-portion
    tray389 tray32 tray189 tray50 - tray
    table144 table229 table30 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray389 kitchen)
     (at tray32 kitchen)
     (at tray189 kitchen)
     (at tray50 kitchen)
     (at_kitchen_bread bread33)
     (at_kitchen_bread bread202)
     (at_kitchen_bread bread391)
     (at_kitchen_content content65)
     (at_kitchen_content content196)
     (at_kitchen_content content470)
     (not_allergic_gluten child49)
     (not_allergic_gluten child266)
     (not_allergic_gluten child251)
     (waiting child251 table30)
     (waiting child266 table30)
     (waiting child49 table229)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child251)
     (served child266)
     (served child49)
    )
  )
)
