; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 506295

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child478 child453 - child
    bread438 bread286 - bread-portion
    content208 content473 - content-portion
    tray386 tray338 tray65 - tray
    table24 table74 table256 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray386 kitchen)
     (at tray338 kitchen)
     (at tray65 kitchen)
     (at_kitchen_bread bread438)
     (at_kitchen_bread bread286)
     (at_kitchen_content content208)
     (at_kitchen_content content473)
     (not_allergic_gluten child453)
     (not_allergic_gluten child478)
     (waiting child478 table256)
     (waiting child453 table24)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child478)
     (served child453)
    )
  )
)
