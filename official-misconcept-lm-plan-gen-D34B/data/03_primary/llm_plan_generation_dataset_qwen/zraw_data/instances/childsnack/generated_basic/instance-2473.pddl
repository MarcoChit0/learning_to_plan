; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 556107

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child158 child269 - child
    bread191 bread108 - bread-portion
    content236 content258 - content-portion
    tray363 tray342 - tray
    table387 table243 table165 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at tray342 kitchen)
     (at_kitchen_bread bread191)
     (at_kitchen_bread bread108)
     (at_kitchen_content content236)
     (at_kitchen_content content258)
     (not_allergic_gluten child269)
     (not_allergic_gluten child158)
     (waiting child158 table165)
     (waiting child269 table165)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child158)
     (served child269)
    )
  )
)
