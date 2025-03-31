; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 810054

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child317 child74 - child
    bread209 bread72 - bread-portion
    content67 content268 - content-portion
    tray431 tray410 - tray
    table98 table319 table326 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray431 kitchen)
     (at tray410 kitchen)
     (at_kitchen_bread bread209)
     (at_kitchen_bread bread72)
     (at_kitchen_content content67)
     (at_kitchen_content content268)
     (no_gluten_bread bread72)
     (no_gluten_content content268)
     (allergic_gluten child317)
     (not_allergic_gluten child74)
     (waiting child317 table326)
     (waiting child74 table98)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child317)
     (served child74)
    )
  )
)
