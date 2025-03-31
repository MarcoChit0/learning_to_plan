; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 144777

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child194 child302 - child
    bread62 bread348 - bread-portion
    content376 content112 - content-portion
    tray227 tray136 tray224 tray79 - tray
    table157 table346 table60 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray227 kitchen)
     (at tray136 kitchen)
     (at tray224 kitchen)
     (at tray79 kitchen)
     (at_kitchen_bread bread62)
     (at_kitchen_bread bread348)
     (at_kitchen_content content376)
     (at_kitchen_content content112)
     (no_gluten_bread bread62)
     (no_gluten_content content376)
     (allergic_gluten child302)
     (not_allergic_gluten child194)
     (waiting child194 table157)
     (waiting child302 table346)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child194)
     (served child302)
    )
  )
)
