; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 312261

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child238 child227 - child
    bread426 bread38 - bread-portion
    content286 content230 - content-portion
    tray191 tray42 tray373 - tray
    table28 table143 table492 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at tray42 kitchen)
     (at tray373 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread38)
     (at_kitchen_content content286)
     (at_kitchen_content content230)
     (not_allergic_gluten child238)
     (not_allergic_gluten child227)
     (waiting child238 table143)
     (waiting child227 table492)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child238)
     (served child227)
    )
  )
)
