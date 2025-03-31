; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 628909

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child44 child109 - child
    bread379 bread371 - bread-portion
    content43 content172 - content-portion
    tray414 tray175 tray43 - tray
    table461 table100 table445 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray414 kitchen)
     (at tray175 kitchen)
     (at tray43 kitchen)
     (at_kitchen_bread bread379)
     (at_kitchen_bread bread371)
     (at_kitchen_content content43)
     (at_kitchen_content content172)
     (not_allergic_gluten child44)
     (not_allergic_gluten child109)
     (waiting child44 table461)
     (waiting child109 table461)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child44)
     (served child109)
    )
  )
)
