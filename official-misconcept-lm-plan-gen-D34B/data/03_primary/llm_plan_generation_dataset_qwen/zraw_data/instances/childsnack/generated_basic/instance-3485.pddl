; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 419620

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child469 child287 - child
    bread101 bread268 - bread-portion
    content195 content463 - content-portion
    tray368 tray465 - tray
    table70 table63 table280 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray368 kitchen)
     (at tray465 kitchen)
     (at_kitchen_bread bread101)
     (at_kitchen_bread bread268)
     (at_kitchen_content content195)
     (at_kitchen_content content463)
     (no_gluten_bread bread101)
     (no_gluten_content content463)
     (allergic_gluten child469)
     (not_allergic_gluten child287)
     (waiting child469 table63)
     (waiting child287 table280)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child469)
     (served child287)
    )
  )
)
