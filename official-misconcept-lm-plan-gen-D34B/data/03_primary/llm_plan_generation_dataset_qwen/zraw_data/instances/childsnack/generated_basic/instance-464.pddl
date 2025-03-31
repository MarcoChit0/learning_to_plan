; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 128988

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child85 child434 child295 - child
    bread28 bread57 bread32 - bread-portion
    content78 content203 content94 - content-portion
    tray479 - tray
    table183 table320 table97 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray479 kitchen)
     (at_kitchen_bread bread28)
     (at_kitchen_bread bread57)
     (at_kitchen_bread bread32)
     (at_kitchen_content content78)
     (at_kitchen_content content203)
     (at_kitchen_content content94)
     (not_allergic_gluten child85)
     (not_allergic_gluten child295)
     (not_allergic_gluten child434)
     (waiting child85 table97)
     (waiting child434 table97)
     (waiting child295 table183)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child85)
     (served child434)
     (served child295)
    )
  )
)
