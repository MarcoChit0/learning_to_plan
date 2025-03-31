; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 77912

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child427 child428 - child
    bread259 bread268 - bread-portion
    content435 content201 - content-portion
    tray358 - tray
    table380 table33 table119 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray358 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_bread bread268)
     (at_kitchen_content content435)
     (at_kitchen_content content201)
     (no_gluten_bread bread268)
     (no_gluten_content content201)
     (allergic_gluten child428)
     (not_allergic_gluten child427)
     (waiting child427 table119)
     (waiting child428 table33)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child427)
     (served child428)
    )
  )
)
