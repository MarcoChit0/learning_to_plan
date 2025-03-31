; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 710774

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child443 child332 - child
    bread237 bread465 - bread-portion
    content399 content490 - content-portion
    tray380 - tray
    table430 table54 table79 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray380 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_bread bread465)
     (at_kitchen_content content399)
     (at_kitchen_content content490)
     (not_allergic_gluten child443)
     (not_allergic_gluten child332)
     (waiting child443 table54)
     (waiting child332 table54)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child443)
     (served child332)
    )
  )
)
