; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 988152

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child461 child280 - child
    bread337 bread441 - bread-portion
    content455 content137 - content-portion
    tray32 tray245 tray178 - tray
    table333 table158 table402 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray32 kitchen)
     (at tray245 kitchen)
     (at tray178 kitchen)
     (at_kitchen_bread bread337)
     (at_kitchen_bread bread441)
     (at_kitchen_content content455)
     (at_kitchen_content content137)
     (no_gluten_bread bread441)
     (no_gluten_content content137)
     (allergic_gluten child280)
     (not_allergic_gluten child461)
     (waiting child461 table158)
     (waiting child280 table402)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child461)
     (served child280)
    )
  )
)
