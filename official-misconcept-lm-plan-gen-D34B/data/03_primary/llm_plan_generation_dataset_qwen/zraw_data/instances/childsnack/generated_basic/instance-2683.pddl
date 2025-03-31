; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 626083

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child419 child462 child172 - child
    bread160 bread418 bread339 - bread-portion
    content37 content144 content266 - content-portion
    tray226 tray381 tray8 - tray
    table295 table329 table158 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray226 kitchen)
     (at tray381 kitchen)
     (at tray8 kitchen)
     (at_kitchen_bread bread160)
     (at_kitchen_bread bread418)
     (at_kitchen_bread bread339)
     (at_kitchen_content content37)
     (at_kitchen_content content144)
     (at_kitchen_content content266)
     (not_allergic_gluten child462)
     (not_allergic_gluten child419)
     (not_allergic_gluten child172)
     (waiting child419 table295)
     (waiting child462 table295)
     (waiting child172 table329)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child419)
     (served child462)
     (served child172)
    )
  )
)
