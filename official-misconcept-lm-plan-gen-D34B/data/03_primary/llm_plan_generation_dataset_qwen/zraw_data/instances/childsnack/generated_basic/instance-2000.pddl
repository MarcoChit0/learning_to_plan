; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 838203

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child370 child268 - child
    bread69 bread195 - bread-portion
    content27 content100 - content-portion
    tray40 - tray
    table35 table231 table142 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray40 kitchen)
     (at_kitchen_bread bread69)
     (at_kitchen_bread bread195)
     (at_kitchen_content content27)
     (at_kitchen_content content100)
     (no_gluten_bread bread69)
     (no_gluten_content content100)
     (allergic_gluten child370)
     (not_allergic_gluten child268)
     (waiting child370 table35)
     (waiting child268 table35)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child370)
     (served child268)
    )
  )
)
