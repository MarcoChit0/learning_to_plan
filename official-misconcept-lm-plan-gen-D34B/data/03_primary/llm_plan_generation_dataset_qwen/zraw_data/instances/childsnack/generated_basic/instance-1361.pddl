; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 906748

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child469 child344 - child
    bread357 bread353 - bread-portion
    content363 content188 - content-portion
    tray398 tray168 tray137 - tray
    table80 table339 table227 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray398 kitchen)
     (at tray168 kitchen)
     (at tray137 kitchen)
     (at_kitchen_bread bread357)
     (at_kitchen_bread bread353)
     (at_kitchen_content content363)
     (at_kitchen_content content188)
     (not_allergic_gluten child344)
     (not_allergic_gluten child469)
     (waiting child469 table227)
     (waiting child344 table339)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child469)
     (served child344)
    )
  )
)
