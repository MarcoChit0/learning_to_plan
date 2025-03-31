; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 226332

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 child317 - child
    bread345 bread239 - bread-portion
    content258 content240 - content-portion
    tray144 tray110 tray342 tray117 - tray
    table195 table254 table468 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray144 kitchen)
     (at tray110 kitchen)
     (at tray342 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread345)
     (at_kitchen_bread bread239)
     (at_kitchen_content content258)
     (at_kitchen_content content240)
     (not_allergic_gluten child317)
     (not_allergic_gluten child73)
     (waiting child73 table468)
     (waiting child317 table195)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child73)
     (served child317)
    )
  )
)
