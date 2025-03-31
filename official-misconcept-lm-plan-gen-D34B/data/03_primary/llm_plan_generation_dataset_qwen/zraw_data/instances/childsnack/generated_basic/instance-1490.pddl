; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 109533

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child303 child67 - child
    bread280 bread198 - bread-portion
    content392 content137 - content-portion
    tray288 tray79 tray103 - tray
    table288 table263 table215 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray288 kitchen)
     (at tray79 kitchen)
     (at tray103 kitchen)
     (at_kitchen_bread bread280)
     (at_kitchen_bread bread198)
     (at_kitchen_content content392)
     (at_kitchen_content content137)
     (no_gluten_bread bread280)
     (no_gluten_content content137)
     (allergic_gluten child67)
     (not_allergic_gluten child303)
     (waiting child303 table215)
     (waiting child67 table215)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child303)
     (served child67)
    )
  )
)
