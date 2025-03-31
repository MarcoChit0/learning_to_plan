; child-snack task with 2 children and 0.51 gluten factor 
; constant factor of 1.9
; random seed: 65563

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child445 child19 - child
    bread250 bread99 - bread-portion
    content407 content276 - content-portion
    tray486 tray3 tray166 - tray
    table62 table185 table151 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray486 kitchen)
     (at tray3 kitchen)
     (at tray166 kitchen)
     (at_kitchen_bread bread250)
     (at_kitchen_bread bread99)
     (at_kitchen_content content407)
     (at_kitchen_content content276)
     (no_gluten_bread bread250)
     (no_gluten_content content407)
     (allergic_gluten child19)
     (not_allergic_gluten child445)
     (waiting child445 table62)
     (waiting child19 table151)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child445)
     (served child19)
    )
  )
)
