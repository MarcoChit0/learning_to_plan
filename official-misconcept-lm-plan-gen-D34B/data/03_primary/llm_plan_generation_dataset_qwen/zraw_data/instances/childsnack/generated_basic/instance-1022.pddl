; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 58147

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child417 child289 child385 - child
    bread324 bread495 bread442 - bread-portion
    content361 content44 content112 - content-portion
    tray229 - tray
    table72 table498 table216 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray229 kitchen)
     (at_kitchen_bread bread324)
     (at_kitchen_bread bread495)
     (at_kitchen_bread bread442)
     (at_kitchen_content content361)
     (at_kitchen_content content44)
     (at_kitchen_content content112)
     (not_allergic_gluten child289)
     (not_allergic_gluten child417)
     (not_allergic_gluten child385)
     (waiting child417 table498)
     (waiting child289 table72)
     (waiting child385 table498)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child417)
     (served child289)
     (served child385)
    )
  )
)
