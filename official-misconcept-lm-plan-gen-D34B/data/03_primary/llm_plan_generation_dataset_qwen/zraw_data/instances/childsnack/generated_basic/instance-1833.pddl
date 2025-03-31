; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 441773

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child156 child483 - child
    bread139 bread206 - bread-portion
    content399 content174 - content-portion
    tray493 - tray
    table481 table207 table214 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray493 kitchen)
     (at_kitchen_bread bread139)
     (at_kitchen_bread bread206)
     (at_kitchen_content content399)
     (at_kitchen_content content174)
     (no_gluten_bread bread206)
     (no_gluten_content content399)
     (allergic_gluten child156)
     (not_allergic_gluten child483)
     (waiting child156 table481)
     (waiting child483 table214)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child156)
     (served child483)
    )
  )
)
