; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 83509

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child78 child273 - child
    bread429 bread414 - bread-portion
    content171 content400 - content-portion
    tray320 tray218 tray173 tray99 - tray
    table28 table274 table195 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray320 kitchen)
     (at tray218 kitchen)
     (at tray173 kitchen)
     (at tray99 kitchen)
     (at_kitchen_bread bread429)
     (at_kitchen_bread bread414)
     (at_kitchen_content content171)
     (at_kitchen_content content400)
     (not_allergic_gluten child273)
     (not_allergic_gluten child78)
     (waiting child78 table274)
     (waiting child273 table274)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child78)
     (served child273)
    )
  )
)
