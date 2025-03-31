; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 618975

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child129 child183 - child
    bread92 bread439 - bread-portion
    content96 content362 - content-portion
    tray476 tray273 tray371 tray387 - tray
    table86 table324 table367 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray476 kitchen)
     (at tray273 kitchen)
     (at tray371 kitchen)
     (at tray387 kitchen)
     (at_kitchen_bread bread92)
     (at_kitchen_bread bread439)
     (at_kitchen_content content96)
     (at_kitchen_content content362)
     (no_gluten_bread bread92)
     (no_gluten_content content362)
     (allergic_gluten child183)
     (not_allergic_gluten child129)
     (waiting child129 table367)
     (waiting child183 table86)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child129)
     (served child183)
    )
  )
)
