; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 444794

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child302 child439 - child
    bread484 bread495 - bread-portion
    content331 content102 - content-portion
    tray241 tray380 tray498 - tray
    table110 table20 table334 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at tray380 kitchen)
     (at tray498 kitchen)
     (at_kitchen_bread bread484)
     (at_kitchen_bread bread495)
     (at_kitchen_content content331)
     (at_kitchen_content content102)
     (not_allergic_gluten child302)
     (not_allergic_gluten child439)
     (waiting child302 table334)
     (waiting child439 table334)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child302)
     (served child439)
    )
  )
)
