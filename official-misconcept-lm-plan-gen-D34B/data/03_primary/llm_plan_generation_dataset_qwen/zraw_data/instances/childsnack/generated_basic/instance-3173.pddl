; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 126255

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child317 child398 - child
    bread164 bread188 - bread-portion
    content189 content103 - content-portion
    tray356 tray224 - tray
    table245 table324 table59 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray356 kitchen)
     (at tray224 kitchen)
     (at_kitchen_bread bread164)
     (at_kitchen_bread bread188)
     (at_kitchen_content content189)
     (at_kitchen_content content103)
     (not_allergic_gluten child398)
     (not_allergic_gluten child317)
     (waiting child317 table59)
     (waiting child398 table245)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child317)
     (served child398)
    )
  )
)
