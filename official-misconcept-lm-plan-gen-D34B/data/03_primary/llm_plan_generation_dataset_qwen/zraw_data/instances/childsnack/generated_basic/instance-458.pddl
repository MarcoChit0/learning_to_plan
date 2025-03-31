; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 801382

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child264 child487 child338 - child
    bread259 bread337 bread327 - bread-portion
    content266 content275 content281 - content-portion
    tray169 - tray
    table291 table171 table371 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray169 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_bread bread337)
     (at_kitchen_bread bread327)
     (at_kitchen_content content266)
     (at_kitchen_content content275)
     (at_kitchen_content content281)
     (not_allergic_gluten child264)
     (not_allergic_gluten child338)
     (not_allergic_gluten child487)
     (waiting child264 table291)
     (waiting child487 table291)
     (waiting child338 table171)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child264)
     (served child487)
     (served child338)
    )
  )
)
