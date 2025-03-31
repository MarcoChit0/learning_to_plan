; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 435216

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child21 child444 - child
    bread226 bread206 - bread-portion
    content258 content472 - content-portion
    tray164 tray427 tray253 - tray
    table245 table33 table237 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray164 kitchen)
     (at tray427 kitchen)
     (at tray253 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_bread bread206)
     (at_kitchen_content content258)
     (at_kitchen_content content472)
     (no_gluten_bread bread226)
     (no_gluten_content content258)
     (allergic_gluten child21)
     (not_allergic_gluten child444)
     (waiting child21 table237)
     (waiting child444 table237)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child21)
     (served child444)
    )
  )
)
