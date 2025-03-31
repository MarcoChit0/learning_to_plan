; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 771864

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child441 child167 - child
    bread166 bread99 - bread-portion
    content63 content217 - content-portion
    tray438 tray62 tray213 - tray
    table46 table404 table463 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray438 kitchen)
     (at tray62 kitchen)
     (at tray213 kitchen)
     (at_kitchen_bread bread166)
     (at_kitchen_bread bread99)
     (at_kitchen_content content63)
     (at_kitchen_content content217)
     (not_allergic_gluten child441)
     (not_allergic_gluten child167)
     (waiting child441 table46)
     (waiting child167 table404)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child441)
     (served child167)
    )
  )
)
