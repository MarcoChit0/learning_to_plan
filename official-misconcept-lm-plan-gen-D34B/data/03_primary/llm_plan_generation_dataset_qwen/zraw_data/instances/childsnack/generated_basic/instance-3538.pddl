; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 873207

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child281 child469 - child
    bread364 bread138 - bread-portion
    content269 content201 - content-portion
    tray410 tray6 - tray
    table298 table7 table65 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray410 kitchen)
     (at tray6 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_bread bread138)
     (at_kitchen_content content269)
     (at_kitchen_content content201)
     (not_allergic_gluten child469)
     (not_allergic_gluten child281)
     (waiting child281 table298)
     (waiting child469 table7)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child281)
     (served child469)
    )
  )
)
