; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 617730

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child344 child444 - child
    bread211 bread75 - bread-portion
    content122 content165 - content-portion
    tray137 - tray
    table70 table60 table220 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray137 kitchen)
     (at_kitchen_bread bread211)
     (at_kitchen_bread bread75)
     (at_kitchen_content content122)
     (at_kitchen_content content165)
     (no_gluten_bread bread75)
     (no_gluten_content content122)
     (allergic_gluten child444)
     (not_allergic_gluten child344)
     (waiting child344 table220)
     (waiting child444 table70)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child344)
     (served child444)
    )
  )
)
