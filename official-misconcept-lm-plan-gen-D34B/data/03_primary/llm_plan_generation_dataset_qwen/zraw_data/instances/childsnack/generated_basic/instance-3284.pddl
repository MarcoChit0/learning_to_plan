; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 758050

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child294 child488 - child
    bread435 bread71 - bread-portion
    content421 content199 - content-portion
    tray263 tray159 tray137 tray87 - tray
    table439 table265 table314 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray263 kitchen)
     (at tray159 kitchen)
     (at tray137 kitchen)
     (at tray87 kitchen)
     (at_kitchen_bread bread435)
     (at_kitchen_bread bread71)
     (at_kitchen_content content421)
     (at_kitchen_content content199)
     (no_gluten_bread bread71)
     (no_gluten_content content421)
     (allergic_gluten child294)
     (not_allergic_gluten child488)
     (waiting child294 table265)
     (waiting child488 table439)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child294)
     (served child488)
    )
  )
)
