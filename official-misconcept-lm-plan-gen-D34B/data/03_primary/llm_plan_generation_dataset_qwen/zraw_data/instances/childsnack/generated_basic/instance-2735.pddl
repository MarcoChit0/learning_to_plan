; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 309578

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child363 child424 - child
    bread165 bread300 - bread-portion
    content419 content392 - content-portion
    tray461 - tray
    table401 table207 table292 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray461 kitchen)
     (at_kitchen_bread bread165)
     (at_kitchen_bread bread300)
     (at_kitchen_content content419)
     (at_kitchen_content content392)
     (no_gluten_bread bread300)
     (no_gluten_content content419)
     (allergic_gluten child424)
     (not_allergic_gluten child363)
     (waiting child363 table292)
     (waiting child424 table401)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child363)
     (served child424)
    )
  )
)
