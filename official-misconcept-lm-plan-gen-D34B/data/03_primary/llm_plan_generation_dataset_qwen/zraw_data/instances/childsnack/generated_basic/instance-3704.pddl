; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 466923

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child294 child244 - child
    bread303 bread329 - bread-portion
    content294 content352 - content-portion
    tray32 tray95 - tray
    table205 table176 table295 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray32 kitchen)
     (at tray95 kitchen)
     (at_kitchen_bread bread303)
     (at_kitchen_bread bread329)
     (at_kitchen_content content294)
     (at_kitchen_content content352)
     (no_gluten_bread bread303)
     (no_gluten_content content294)
     (allergic_gluten child294)
     (not_allergic_gluten child244)
     (waiting child294 table295)
     (waiting child244 table205)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child294)
     (served child244)
    )
  )
)
