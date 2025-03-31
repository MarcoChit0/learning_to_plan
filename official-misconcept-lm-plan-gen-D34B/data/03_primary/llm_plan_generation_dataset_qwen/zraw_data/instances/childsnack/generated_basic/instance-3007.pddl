; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 517156

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child497 child198 - child
    bread205 bread383 - bread-portion
    content313 content215 - content-portion
    tray414 tray465 tray257 tray488 - tray
    table325 table78 table348 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray414 kitchen)
     (at tray465 kitchen)
     (at tray257 kitchen)
     (at tray488 kitchen)
     (at_kitchen_bread bread205)
     (at_kitchen_bread bread383)
     (at_kitchen_content content313)
     (at_kitchen_content content215)
     (not_allergic_gluten child497)
     (not_allergic_gluten child198)
     (waiting child497 table348)
     (waiting child198 table325)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child497)
     (served child198)
    )
  )
)
