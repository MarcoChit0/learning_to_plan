; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 441497

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child422 child472 - child
    bread324 bread493 - bread-portion
    content371 content434 - content-portion
    tray250 tray285 - tray
    table46 table473 table193 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray250 kitchen)
     (at tray285 kitchen)
     (at_kitchen_bread bread324)
     (at_kitchen_bread bread493)
     (at_kitchen_content content371)
     (at_kitchen_content content434)
     (not_allergic_gluten child472)
     (not_allergic_gluten child422)
     (waiting child422 table46)
     (waiting child472 table473)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child422)
     (served child472)
    )
  )
)
