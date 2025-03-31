; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 171088

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child104 - child
    bread433 - bread-portion
    content473 - content-portion
    tray76 tray435 tray139 - tray
    table92 table353 table366 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray76 kitchen)
     (at tray435 kitchen)
     (at tray139 kitchen)
     (at_kitchen_bread bread433)
     (at_kitchen_content content473)
     (not_allergic_gluten child104)
     (waiting child104 table92)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child104)
    )
  )
)
