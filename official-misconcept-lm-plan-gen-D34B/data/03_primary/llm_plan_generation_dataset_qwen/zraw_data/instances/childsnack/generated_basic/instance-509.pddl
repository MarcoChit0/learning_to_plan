; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 748406

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child430 child21 - child
    bread122 bread65 - bread-portion
    content98 content337 - content-portion
    tray413 tray22 - tray
    table99 table399 table223 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray413 kitchen)
     (at tray22 kitchen)
     (at_kitchen_bread bread122)
     (at_kitchen_bread bread65)
     (at_kitchen_content content98)
     (at_kitchen_content content337)
     (not_allergic_gluten child21)
     (not_allergic_gluten child430)
     (waiting child430 table399)
     (waiting child21 table399)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child430)
     (served child21)
    )
  )
)
